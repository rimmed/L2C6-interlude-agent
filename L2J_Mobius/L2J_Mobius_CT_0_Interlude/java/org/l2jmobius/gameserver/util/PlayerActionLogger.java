/*
 * Copyright (c) 2013 L2jMobius
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be
 * included in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
 * WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR
 * IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */
package org.l2jmobius.gameserver.util;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Queue;
import java.util.concurrent.ConcurrentHashMap;
import java.util.logging.Handler;
import java.util.logging.Logger;

import org.l2jmobius.gameserver.config.custom.RLAgentEnvironmentConfig;
import org.l2jmobius.gameserver.geoengine.GeoEngine;
import org.l2jmobius.gameserver.managers.RLAgentManager;
import org.l2jmobius.gameserver.model.World;
import org.l2jmobius.gameserver.model.WorldObject;
import org.l2jmobius.gameserver.model.actor.Attackable;
import org.l2jmobius.gameserver.model.actor.Creature;
import org.l2jmobius.gameserver.model.actor.Npc;
import org.l2jmobius.gameserver.model.actor.Player;
import org.l2jmobius.gameserver.model.actor.Summon;
import org.l2jmobius.gameserver.model.actor.holders.creature.TimeStamp;
import org.l2jmobius.gameserver.model.actor.holders.npc.AggroInfo;
import org.l2jmobius.gameserver.model.item.instance.Item;
import org.l2jmobius.gameserver.model.itemcontainer.Inventory;
import org.l2jmobius.gameserver.model.skill.BuffInfo;
import org.l2jmobius.gameserver.model.skill.Skill;

/**
 * Player action log (format 2): one compact, human-readable line per action.
 * <p>
 * {@code <date> <time> <character> <EVENT> <payload> [| me ...] [| sel ...] [| env ...] [| cd ...]}
 * <ul>
 * <li>references: {@code Name#objectId(npcId Llvl) hp cur/max} for monsters and NPCs, {@code Name#objectId(pc Llvl) hp cur/max} for players, {@code Name#objectId(item itemId)} for items, {@code -} for none;</li>
 * <li>{@code me L18 hp 711/711 mp 231/231 cp 368/368 @x,y,z h<heading> wt<load %>} plus the flags {@code sit cast atk dead combat} when set;</li>
 * <li>{@code sel d<distance> los|nolos h<target heading>}: the selected target, relative to the character;</li>
 * <li>{@code pet Name#objectId(summon npcId Llvl) hp cur/max mp cur/max @x,y,z d<distance> tgt <objectId>|-} plus the flags {@code atk cast dead follow}: the character's summon (servitor or pet), when it has one;</li>
 * <li>{@code env items <ground items within 1000>; <npcId>#<objectId> L<lvl> d<distance> hp<hp %> [by:<players>]; ...}: the 10 nearest monsters within 3000, {@code by:} lists the other players on a monster's aggro list;</li>
 * <li>{@code cd <skillId>:<ms>,...}: skill reuse still running.</li>
 * </ul>
 * RL agents are not logged.
 * A summon's actions are written on its owner's lines: {@code SUMMON} / {@code UNSUMMON} / {@code PETDIED}, the owner's commands ({@code PETCMD}), its casts ({@code PETSKILL}), its damage ({@code DMG ... by pet}), its kills ({@code KILL ... by pet}) and the damage it takes ({@code PETHIT}).
 * A kill is one {@code KILL} line with every drop ({@code drop Name[count] auto|ground; ...}, {@code drop -} without drops).
 * Character state that rarely changes is written on its own lines, only when it changed: {@code STATS} (level, exp, sp and combat stats), {@code EQUIP} (paperdoll) and {@code SKILLS} (known skills). A parser keeps the last state per character.
 */
public class PlayerActionLogger
{
	private static final Logger LOGGER = Logger.getLogger("playeraction");
	private static final long BUFF_DEBUFF_LOG_COOLDOWN = 5000; // 5 seconds
	private static final Map<Integer, Long> _buffDebuffLogTimes = new ConcurrentHashMap<>();
	private static final long SKILLS_LOG_COOLDOWN = 30000; // 30 seconds
	private static final Map<Integer, Long> _skillsLogTimes = new ConcurrentHashMap<>();
	private static final long DAMAGE_LOG_COOLDOWN = 100; // 100 ms dedup window
	private static final Map<Long, Long> _damageLogTimes = new ConcurrentHashMap<>();
	private static final Map<Long, Long> _petHitLogTimes = new ConcurrentHashMap<>();
	private static final long TARGET_SELECT_LOG_COOLDOWN = 1000; // 1s dedup window for target select
	private static final Map<Long, Long> _targetSelectLogTimes = new ConcurrentHashMap<>();
	
	/** Monsters listed in the {@code env} block: the nearest ones within this range. */
	private static final int ENV_RANGE = 3000;
	private static final int ENV_MAX_MONSTERS = 10;
	private static final int GROUND_ITEM_RANGE = 1000;
	
	/** RequestActionUse actions that command a summon; the other summon actions are its skills (logged as {@code PETCMD skill<actionId>}, the cast itself as {@code PETSKILL}). */
	private static final Map<Integer, String> SUMMON_COMMANDS = Map.of(15, "follow", 21, "follow", 16, "attack", 22, "attack", 17, "stop", 23, "stop", 19, "unsummon", 52, "unsummon", 53, "move", 54, "move");

	/** Last {@code STATS} / {@code EQUIP} / {@code SKILLS} text written per character, so a line is only written on change. */
	private static final Map<Integer, String> _lastStats = new ConcurrentHashMap<>();
	private static final Map<Integer, String> _lastEquip = new ConcurrentHashMap<>();
	private static final Map<Integer, String> _lastSkills = new ConcurrentHashMap<>();
	
	private static final String[] SLOT_NAMES =
	{
		"weapon",
		"shield",
		"head",
		"chest",
		"gloves",
		"legs",
		"feet",
		"neck",
		"rear",
		"lear",
		"rfinger",
		"lfinger"
	};
	private static final int[] SLOTS =
	{
		Inventory.PAPERDOLL_RHAND,
		Inventory.PAPERDOLL_LHAND,
		Inventory.PAPERDOLL_HEAD,
		Inventory.PAPERDOLL_CHEST,
		Inventory.PAPERDOLL_GLOVES,
		Inventory.PAPERDOLL_LEGS,
		Inventory.PAPERDOLL_FEET,
		Inventory.PAPERDOLL_NECK,
		Inventory.PAPERDOLL_REAR,
		Inventory.PAPERDOLL_LEAR,
		Inventory.PAPERDOLL_RFINGER,
		Inventory.PAPERDOLL_LFINGER
	};
	
	static
	{
		Runtime.getRuntime().addShutdownHook(new Thread(PlayerActionLogger::flush, "PlayerActionLogger-ShutdownHook"));
	}
	
	public static void flush()
	{
		for (Handler handler : LOGGER.getHandlers())
		{
			handler.flush();
		}
	}
	
	/** RL agents are not logged: hundreds of them would flood the log, and their trajectories go through the RL pipeline. */
	private static boolean isAgent(Player player)
	{
		return RLAgentEnvironmentConfig.RL_AGENT_ENABLED && RLAgentManager.getInstance().isAgent(player);
	}
	
	// ========================================================================
	// Line parts
	// ========================================================================
	
	/** {@code Name#objectId(npcId Llvl) hp cur/max}, {@code Name#objectId(pc Llvl) hp cur/max}, {@code Name#objectId(summon npcId Llvl) hp cur/max}, {@code Name#objectId(item itemId)} or {@code -}. */
	private static String ref(WorldObject object)
	{
		if (object == null)
		{
			return "-";
		}
		
		final StringBuilder sb = new StringBuilder(64);
		sb.append(object.getName()).append('#').append(object.getObjectId());
		if (object.isNpc())
		{
			final Npc npc = (Npc) object;
			sb.append('(').append(npc.getId()).append(" L").append(npc.getLevel()).append(") hp ").append((int) npc.getCurrentHp()).append('/').append(npc.getMaxHp());
		}
		else if (object.isPlayer())
		{
			final Player player = object.asPlayer();
			sb.append("(pc L").append(player.getLevel()).append(") hp ").append((int) player.getCurrentHp()).append('/').append(player.getMaxHp());
		}
		else if (object.isSummon())
		{
			final Summon summon = object.asSummon();
			sb.append("(summon ").append(summon.getId()).append(" L").append(summon.getLevel()).append(") hp ").append((int) summon.getCurrentHp()).append('/').append(summon.getMaxHp());
		}
		else if (object.isItem())
		{
			sb.append("(item ").append(((Item) object).getId()).append(')');
		}
		else if (object instanceof Creature)
		{
			final Creature creature = (Creature) object;
			sb.append("(L").append(creature.getLevel()).append(") hp ").append((int) creature.getCurrentHp()).append('/').append(creature.getMaxHp());
		}
		return sb.toString();
	}
	
	private static String me(Player player, boolean casting)
	{
		final StringBuilder sb = new StringBuilder(128);
		sb.append("me L").append(player.getLevel());
		sb.append(" hp ").append((int) player.getCurrentHp()).append('/').append(player.getMaxHp());
		sb.append(" mp ").append((int) player.getCurrentMp()).append('/').append(player.getMaxMp());
		sb.append(" cp ").append((int) player.getCurrentCp()).append('/').append(player.getMaxCp());
		sb.append(" @").append(player.getX()).append(',').append(player.getY()).append(',').append(player.getZ());
		sb.append(" h").append(player.getHeading());
		sb.append(" wt").append(String.format("%.1f", (player.getMaxLoad() > 0) ? ((double) player.getCurrentLoad() / player.getMaxLoad()) * 100 : 0));
		if (player.isSitting())
		{
			sb.append(" sit");
		}
		if (casting)
		{
			sb.append(" cast");
		}
		if (player.isAttackingNow())
		{
			sb.append(" atk");
		}
		if (player.isAlikeDead())
		{
			sb.append(" dead");
		}
		if (player.isInCombat())
		{
			sb.append(" combat");
		}
		return sb.toString();
	}
	
	/** The selected target relative to the character, or {@code null} without one. */
	private static String sel(Player player, WorldObject target)
	{
		if (target == null)
		{
			return null;
		}
		
		final StringBuilder sb = new StringBuilder(32);
		sb.append("sel d").append(String.format("%.1f", player.calculateDistance3D(target)));
		sb.append(player.isInsideRadius3D(target, 1500) && GeoEngine.getInstance().canSeeTarget(player, target) ? " los" : " nolos");
		if (target instanceof Creature)
		{
			sb.append(" h").append(((Creature) target).getHeading());
		}
		return sb.toString();
	}
	
	/** The character's summon, or {@code null} without one. */
	private static String pet(Player player)
	{
		final Summon summon = player.getSummon();
		if (summon == null)
		{
			return null;
		}

		final StringBuilder sb = new StringBuilder(128);
		sb.append("pet ").append(ref(summon));
		sb.append(" mp ").append((int) summon.getCurrentMp()).append('/').append(summon.getMaxMp());
		sb.append(" @").append(summon.getX()).append(',').append(summon.getY()).append(',').append(summon.getZ());
		sb.append(" d").append((int) player.calculateDistance3D(summon));
		final WorldObject target = summon.getTarget();
		sb.append(" tgt ").append((target != null) ? String.valueOf(target.getObjectId()) : "-");
		if (summon.isAttackingNow())
		{
			sb.append(" atk");
		}
		if (summon.isCastingNow())
		{
			sb.append(" cast");
		}
		if (summon.isDead())
		{
			sb.append(" dead");
		}
		if (summon.getFollowStatus())
		{
			sb.append(" follow");
		}
		return sb.toString();
	}

	private static String env(Player player)
	{
		final StringBuilder sb = new StringBuilder(256);
		sb.append("env items ").append(World.getInstance().getVisibleObjectsInRange(player, Item.class, GROUND_ITEM_RANGE).size());
		
		final List<Attackable> monsters = new ArrayList<>(World.getInstance().getVisibleObjectsInRange(player, Attackable.class, ENV_RANGE));
		monsters.sort((a, b) -> Double.compare(player.calculateDistance3D(a), player.calculateDistance3D(b)));
		for (int i = 0; (i < monsters.size()) && (i < ENV_MAX_MONSTERS); i++)
		{
			final Attackable monster = monsters.get(i);
			sb.append("; ").append(monster.getId()).append('#').append(monster.getObjectId());
			sb.append(" L").append(monster.getLevel());
			sb.append(" d").append((int) player.calculateDistance3D(monster));
			sb.append(" hp").append((monster.getMaxHp() > 0) ? (int) Math.round((monster.getCurrentHp() * 100) / monster.getMaxHp()) : 0);
			final String others = otherAttackers(monster, player);
			if (!others.isEmpty())
			{
				sb.append(" by:").append(others);
			}
		}
		return sb.toString();
	}
	
	/** Players (or the owners of their summons) on the monster's aggro list, other than {@code self}. */
	private static String otherAttackers(Attackable monster, Player self)
	{
		final StringBuilder sb = new StringBuilder();
		for (AggroInfo info : monster.getAggroList().values())
		{
			final Creature attacker = info.getAttacker();
			final Player player = attacker.isPlayer() ? attacker.asPlayer() : attacker.isSummon() ? attacker.asSummon().getOwner() : null;
			if ((player == null) || (player == self) || ((info.getDamage() <= 0) && (info.getHate() <= 0)))
			{
				continue;
			}
			if (sb.length() > 0)
			{
				sb.append(',');
			}
			sb.append(player.getName());
		}
		return sb.toString();
	}
	
	/** Skill reuse still running, or {@code null}. */
	private static String cd(Player player)
	{
		final Map<Integer, TimeStamp> stamps = player.getSkillReuseTimeStamps();
		if ((stamps == null) || stamps.isEmpty())
		{
			return null;
		}
		
		final StringBuilder sb = new StringBuilder("cd ");
		boolean first = true;
		for (TimeStamp stamp : stamps.values())
		{
			final long remaining = stamp.getRemaining();
			if (remaining > 0)
			{
				if (!first)
				{
					sb.append(',');
				}
				sb.append(stamp.getSkillId()).append(':').append(remaining);
				first = false;
			}
		}
		return first ? null : sb.toString();
	}
	
	/** The {@code | me | sel | env | cd} tail of an action line. */
	private static String tail(Player player, WorldObject target, boolean casting)
	{
		final StringBuilder sb = new StringBuilder(512);
		sb.append(" | ").append(me(player, casting));
		final String sel = sel(player, target);
		if (sel != null)
		{
			sb.append(" | ").append(sel);
		}
		final String pet = pet(player);
		if (pet != null)
		{
			sb.append(" | ").append(pet);
		}
		sb.append(" | ").append(env(player));
		final String cd = cd(player);
		if (cd != null)
		{
			sb.append(" | ").append(cd);
		}
		return sb.toString();
	}
	
	private static String stats(Player player)
	{
		return "L" + player.getLevel() + " exp " + player.getExp() + " sp " + player.getSp() + " patk " + (int) player.getPAtk(player) + " matk " + (int) player.getMAtk(player, null) + " pdef " + (int) player.getPDef(player) + " mdef " + (int) player.getMDef(player, null) + " acc " + player.getAccuracy() + " crit " + player.getCriticalHit(player, null) + " eva " + player.getEvasionRate(player) + " aspd " + player.getPAtkSpd() + " cspd " + player.getMAtkSpd() + " str " + player.getSTR() + " int " + player.getINT() + " dex " + player.getDEX() + " con " + player.getCON() + " wit " + player.getWIT();
	}
	
	private static String equipment(Player player)
	{
		final StringBuilder sb = new StringBuilder(256);
		for (int i = 0; i < SLOTS.length; i++)
		{
			final Item item = player.getInventory().getPaperdollItem(SLOTS[i]);
			if (i > 0)
			{
				sb.append(' ');
			}
			sb.append(SLOT_NAMES[i]).append(' ');
			if (item == null)
			{
				sb.append('-');
			}
			else
			{
				sb.append(item.getTemplate().getName()).append('[').append(item.getId()).append(']');
			}
		}
		return sb.toString();
	}
	
	private static String skills(Player player)
	{
		final StringBuilder active = new StringBuilder();
		final StringBuilder passive = new StringBuilder();
		for (Skill skill : player.getSkills().values())
		{
			final StringBuilder sb = skill.isPassive() ? passive : active;
			if (sb.length() > 0)
			{
				sb.append(',');
			}
			sb.append(skill.getId()).append('-').append(skill.getLevel());
		}
		return "active " + (active.length() > 0 ? active : "-") + " passive " + (passive.length() > 0 ? passive : "-");
	}
	
	/** Writes the {@code STATS} / {@code EQUIP} lines of a character whose state changed since its last line. */
	private static void state(Player player)
	{
		final String stats = stats(player);
		if (!stats.equals(_lastStats.put(player.getObjectId(), stats)))
		{
			LOGGER.info(player.getName() + " STATS " + stats);
		}
		final String equipment = equipment(player);
		if (!equipment.equals(_lastEquip.put(player.getObjectId(), equipment)))
		{
			LOGGER.info(player.getName() + " EQUIP " + equipment);
		}
	}
	
	/** Writes an action line with its state lines and tail. */
	private static void action(Player player, String event, WorldObject target, boolean casting)
	{
		state(player);
		LOGGER.info(player.getName() + " " + event + tail(player, target, casting));
		flush();
	}
	
	private static String skillRef(Skill skill)
	{
		return (skill == null) ? "-" : skill.getName() + "[" + skill.getId() + "]";
	}
	
	// ========================================================================
	// Public Log Methods
	// ========================================================================
	
	public static void logSkillsUpdated(Player player)
	{
		if ((player == null) || isAgent(player)) return;
		final String skills = skills(player);
		if (!skills.equals(_lastSkills.put(player.getObjectId(), skills)))
		{
			state(player);
			LOGGER.info(player.getName() + " SKILLS " + skills);
			flush();
		}
	}
	
	public static void logTargetSelect(Player player, WorldObject target)
	{
		if ((player == null) || isAgent(player)) return;
		
		final long key = ((long) player.getObjectId() << 32) | (target != null ? target.getObjectId() : 0);
		final Long lastTime = _targetSelectLogTimes.get(key);
		final long now = System.currentTimeMillis();
		if ((lastTime != null) && ((now - lastTime) < TARGET_SELECT_LOG_COOLDOWN)) return;
		_targetSelectLogTimes.put(key, now);
		
		action(player, "TARGET " + ref(target), target, player.isCastingNow());
	}
	
	public static void logMove(Player player, int x, int y, int z)
	{
		if ((player == null) || isAgent(player)) return;
		
		WorldObject target = player.getTarget();
		// Dead targets are left out, so a corpse is not tracked as a target.
		if ((target instanceof Creature) && (((Creature) target).getCurrentHp() <= 0))
		{
			target = null;
		}
		action(player, "MOVE " + x + "," + y + "," + z, target, player.isCastingNow());
	}
	
	public static void logAttack(Player player, WorldObject target)
	{
		if ((player == null) || isAgent(player)) return;
		action(player, "ATTACK " + ref(target), target, player.isCastingNow());
	}
	
	public static void logSkillUse(Player player, int skillId, String skillName, WorldObject target)
	{
		if ((player == null) || isAgent(player)) return;
		action(player, "SKILL " + skillName + "[" + skillId + "] -> " + ref(target), target, player.isCastingNow());
	}
	
	public static void logShotAutoUse(Player player, Item item, boolean enabled)
	{
		if ((player == null) || (item == null) || isAgent(player)) return;
		LOGGER.info(player.getName() + " AUTOSHOT " + (enabled ? "on " : "off ") + item.getTemplate().getName() + "[" + item.getId() + "]");
		flush();
	}
	
	public static void logItemUse(Player player, Item item)
	{
		if ((player == null) || (item == null) || isAgent(player)) return;
		LOGGER.info(player.getName() + " ITEM " + item.getTemplate().getName() + "[" + item.getId() + "] left " + item.getCount());
		flush();
	}
	
	public static void logAction(Player player, String actionName, String details)
	{
		if ((player == null) || isAgent(player)) return;
		
		WorldObject target = player.getTarget();
		final boolean pickup = (actionName != null) && actionName.toLowerCase().contains("pickup");
		if (pickup && (target instanceof Creature) && (((Creature) target).getCurrentHp() <= 0))
		{
			target = null;
		}
		action(player, (pickup ? "PICKUP " : "ACTION " + actionName + " ") + details, target, player.isCastingNow());
	}
	
	/**
	 * One line per kill, on the line of the character that finished the monster ({@code by pet} when its summon did), with every drop or {@code -}.
	 * @param killer the creature that dealt the killing blow
	 * @param victim the monster
	 * @param drops {@code Name[count] auto|ground} of each dropped item
	 */
	public static void logKill(Creature killer, Attackable victim, List<String> drops)
	{
		if ((killer == null) || (victim == null)) return;
		final Player player = killer.asPlayer();
		if ((player == null) || isAgent(player)) return;

		state(player);
		LOGGER.info(player.getName() + " KILL " + ref(victim) + (killer.isSummon() ? " by pet" : "") + " drop " + (drops.isEmpty() ? "-" : String.join("; ", drops)) + tail(player, player.getTarget(), false));
		flush();
	}
	
	public static void logShotUsage(Player player, String shotName, long remainingCount)
	{
		if ((player == null) || isAgent(player)) return;
		LOGGER.info(player.getName() + " SHOT " + shotName + " left " + remainingCount);
		flush();
	}
	
	public static void logDamageTaken(Player player, double damage, Creature attacker, Skill skill)
	{
		if ((player == null) || isAgent(player)) return;
		
		final long key = ((long) player.getObjectId() << 32) | (attacker != null ? attacker.getObjectId() : 0);
		final Long lastTime = _damageLogTimes.get(key);
		final long now = System.currentTimeMillis();
		if ((lastTime != null) && ((now - lastTime) < DAMAGE_LOG_COOLDOWN)) return;
		_damageLogTimes.put(key, now);
		
		action(player, "HIT " + (damage == 0 ? "miss" : String.valueOf((int) damage)) + " <- " + ref(attacker) + " skill " + skillRef(skill), attacker, player.isCastingNow());
	}
	
	public static void logDamageInflicted(Player player, double damage, Creature target, Skill skill)
	{
		if ((player == null) || isAgent(player)) return;
		
		final long key = ((long) player.getObjectId() << 32) | (target != null ? target.getObjectId() : 0);
		final Long lastTime = _damageLogTimes.get(key);
		final long now = System.currentTimeMillis();
		if ((lastTime != null) && ((now - lastTime) < DAMAGE_LOG_COOLDOWN)) return;
		_damageLogTimes.put(key, now);
		
		action(player, "DMG " + (damage == 0 ? "miss" : String.valueOf((int) damage)) + " -> " + ref(target) + " skill " + skillRef(skill), target, false);
	}
	
	public static void logBuffsDebuffs(Player player)
	{
		if ((player == null) || isAgent(player)) return;
		final int playerId = player.getObjectId();
		final Long lastTime = _buffDebuffLogTimes.get(playerId);
		final long now = System.currentTimeMillis();
		if ((lastTime != null) && ((now - lastTime) < BUFF_DEBUFF_LOG_COOLDOWN)) return;
		_buffDebuffLogTimes.put(playerId, now);
		
		action(player, "EFFECTS buffs " + effects(player.getEffectList().getBuffs()) + " debuffs " + effects(player.getEffectList().getDebuffs()), player.getTarget(), player.isCastingNow());
	}
	
	private static String effects(Queue<BuffInfo> infos)
	{
		if (infos.isEmpty())
		{
			return "-";
		}
		final StringBuilder sb = new StringBuilder();
		for (BuffInfo info : infos)
		{
			if (sb.length() > 0)
			{
				sb.append(',');
			}
			sb.append(skillRef(info.getSkill())).append(':').append(info.getTime()).append('s');
		}
		return sb.toString();
	}
	
	public static void logEffectApplied(Player player, BuffInfo info)
	{
		if ((player == null) || (info == null) || (info.getSkill() == null) || isAgent(player)) return;
		final Skill skill = info.getSkill();
		state(player);
		LOGGER.info(player.getName() + " EFFECT " + (skill.isDebuff() ? "debuff " : "buff ") + skillRef(skill) + " " + info.getTime() + "s" + tail(player, player.getTarget(), player.isCastingNow()));
	}
	
	public static void logSkills(Player player)
	{
		if ((player == null) || isAgent(player)) return;
		final int playerId = player.getObjectId();
		final Long lastTime = _skillsLogTimes.get(playerId);
		final long now = System.currentTimeMillis();
		if ((lastTime != null) && ((now - lastTime) < SKILLS_LOG_COOLDOWN)) return;
		_skillsLogTimes.put(playerId, now);
		logSkillsUpdated(player);
	}
	
	// -------------------------------------------------------------------------
	// Summons (servitors and pets), written on the owner's lines
	// -------------------------------------------------------------------------

	/** The owner whose log gets a summon's line, or {@code null} when nothing is logged. */
	private static Player loggedOwner(Summon summon)
	{
		final Player owner = (summon == null) ? null : summon.getOwner();
		return ((owner == null) || isAgent(owner)) ? null : owner;
	}

	/** The owner gave the summon a command (RequestActionUse); actions that are not summon actions are ignored. */
	public static void logSummonCommand(Player player, int actionId, WorldObject target)
	{
		if ((player == null) || (player.getSummon() == null) || isAgent(player)) return;
		String command = SUMMON_COMMANDS.get(actionId);
		if ((command == null) && (((actionId >= 32) && (actionId <= 54) && (actionId != 37) && (actionId != 38) && (actionId != 51)) || (actionId >= 1000)))
		{
			command = "skill" + actionId;
		}
		if (command != null)
		{
			action(player, "PETCMD " + command + " -> " + ref(target), target, player.isCastingNow());
		}
	}

	public static void logSummonSpawn(Summon summon)
	{
		final Player owner = loggedOwner(summon);
		if (owner == null) return;
		action(owner, "SUMMON " + ref(summon), owner.getTarget(), owner.isCastingNow());
	}

	public static void logSummonUnsummon(Summon summon)
	{
		final Player owner = loggedOwner(summon);
		if (owner == null) return;
		action(owner, "UNSUMMON " + ref(summon), owner.getTarget(), owner.isCastingNow());
	}

	public static void logSummonDeath(Summon summon, Creature killer)
	{
		final Player owner = loggedOwner(summon);
		if (owner == null) return;
		action(owner, "PETDIED " + ref(summon) + " <- " + ref(killer), owner.getTarget(), owner.isCastingNow());
	}

	public static void logSummonSkill(Summon summon, Skill skill, WorldObject target)
	{
		final Player owner = loggedOwner(summon);
		if ((owner == null) || (skill == null)) return;
		action(owner, "PETSKILL " + skillRef(skill) + " -> " + ref(target), owner.getTarget(), owner.isCastingNow());
	}

	public static void logSummonDamageInflicted(Summon summon, double damage, Creature target, Skill skill)
	{
		final Player owner = loggedOwner(summon);
		if (owner == null) return;

		final long key = ((long) summon.getObjectId() << 32) | (target != null ? target.getObjectId() : 0);
		final Long lastTime = _damageLogTimes.get(key);
		final long now = System.currentTimeMillis();
		if ((lastTime != null) && ((now - lastTime) < DAMAGE_LOG_COOLDOWN)) return;
		_damageLogTimes.put(key, now);

		action(owner, "DMG " + (damage == 0 ? "miss" : String.valueOf((int) damage)) + " -> " + ref(target) + " skill " + skillRef(skill) + " by pet", owner.getTarget(), owner.isCastingNow());
	}

	public static void logSummonDamageTaken(Summon summon, double damage, Creature attacker, Skill skill)
	{
		final Player owner = loggedOwner(summon);
		if (owner == null) return;

		final long key = ((long) summon.getObjectId() << 32) | (attacker != null ? attacker.getObjectId() : 0);
		final Long lastTime = _petHitLogTimes.get(key);
		final long now = System.currentTimeMillis();
		if ((lastTime != null) && ((now - lastTime) < DAMAGE_LOG_COOLDOWN)) return;
		_petHitLogTimes.put(key, now);

		action(owner, "PETHIT " + (int) damage + " <- " + ref(attacker) + " skill " + skillRef(skill), owner.getTarget(), owner.isCastingNow());
	}

	// -------------------------------------------------------------------------
	// Economy, progression, travel, life state
	// -------------------------------------------------------------------------
	
	public static void logItemPurchase(Player player, String itemName, int itemId, long count, long price, String npcName, int npcId)
	{
		if ((player == null) || isAgent(player)) return;
		LOGGER.info(player.getName() + " BUY " + itemName + "[" + itemId + "] x" + count + " price " + price + " npc " + npcName + "[" + npcId + "]");
		flush();
	}
	
	public static void logItemSell(Player player, String itemName, int itemId, long count, long price, String npcName, int npcId)
	{
		if ((player == null) || isAgent(player)) return;
		LOGGER.info(player.getName() + " SELL " + itemName + "[" + itemId + "] x" + count + " price " + price + " npc " + npcName + "[" + npcId + "]");
		flush();
	}
	
	public static void logSkillLearned(Player player, String skillName, int skillId, int skillLevel, int spCost, long adenaCost)
	{
		if ((player == null) || isAgent(player)) return;
		LOGGER.info(player.getName() + " LEARN " + skillName + "[" + skillId + "] L" + skillLevel + " cost " + spCost + "sp" + (adenaCost > 0 ? " " + adenaCost + "a" : ""));
		flush();
	}
	
	public static void logTeleport(Player player, int fromX, int fromY, int fromZ, int toX, int toY, int toZ, String method)
	{
		if ((player == null) || isAgent(player)) return;
		LOGGER.info(player.getName() + " TELEPORT " + fromX + "," + fromY + "," + fromZ + " -> " + toX + "," + toY + "," + toZ + " via " + method);
		flush();
	}
	
	public static void logDeath(Player player, String killerName, int killerId)
	{
		if ((player == null) || isAgent(player)) return;
		LOGGER.info(player.getName() + " DIED @" + player.getX() + "," + player.getY() + "," + player.getZ() + " killer " + killerName + "[" + killerId + "]");
		flush();
	}
	
	public static void logResurrection(Player player, String method)
	{
		if ((player == null) || isAgent(player)) return;
		LOGGER.info(player.getName() + " REVIVED @" + player.getX() + "," + player.getY() + "," + player.getZ() + " via " + method);
		flush();
	}
}
