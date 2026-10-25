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
package org.l2jmobius.gameserver.managers;

import java.sql.Connection;
import java.sql.PreparedStatement;
import java.sql.ResultSet;
import java.util.ArrayList;
import java.util.Collections;
import java.util.EnumMap;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Map.Entry;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ScheduledFuture;
import java.util.function.Consumer;
import java.util.function.Predicate;
import java.util.logging.Level;
import java.util.logging.Logger;

import org.l2jmobius.commons.database.DatabaseFactory;
import org.l2jmobius.commons.threads.ThreadPool;
import org.l2jmobius.commons.util.Rnd;
import org.l2jmobius.gameserver.config.custom.AllowedPlayerRacesConfig;
import org.l2jmobius.gameserver.config.custom.RLAgentEnvironmentConfig;
import org.l2jmobius.gameserver.config.custom.RLAgentEnvironmentConfig.FarmZone;
import org.l2jmobius.gameserver.config.custom.RLAgentEnvironmentConfig.PotionEntry;
import org.l2jmobius.gameserver.data.sql.CharInfoTable;
import org.l2jmobius.gameserver.data.xml.ExperienceData;
import org.l2jmobius.gameserver.data.xml.ItemData;
import org.l2jmobius.gameserver.data.xml.PlayerTemplateData;
import org.l2jmobius.gameserver.data.xml.SkillData;
import org.l2jmobius.gameserver.model.effects.AbstractEffect;
import org.l2jmobius.gameserver.model.skill.EffectScope;
import org.l2jmobius.gameserver.model.skill.Skill;
import org.l2jmobius.gameserver.model.Location;
import org.l2jmobius.gameserver.model.World;
import org.l2jmobius.gameserver.model.actor.Player;
import org.l2jmobius.gameserver.model.actor.appearance.PlayerAppearance;
import org.l2jmobius.gameserver.model.actor.enums.player.PlayerClass;
import org.l2jmobius.gameserver.model.actor.templates.PlayerTemplate;
import org.l2jmobius.gameserver.model.events.Containers;
import org.l2jmobius.gameserver.model.events.EventType;
import org.l2jmobius.gameserver.model.events.ListenersContainer;
import org.l2jmobius.gameserver.model.events.holders.actor.creature.OnCreatureDeath;
import org.l2jmobius.gameserver.model.events.holders.actor.player.OnPlayerCreate;
import org.l2jmobius.gameserver.model.events.holders.actor.player.OnPlayerLevelChanged;
import org.l2jmobius.gameserver.model.events.holders.actor.player.OnPlayerLogin;
import org.l2jmobius.gameserver.model.events.holders.actor.player.OnPlayerLogout;
import org.l2jmobius.gameserver.model.events.listeners.ConsumerEventListener;
import org.l2jmobius.gameserver.model.itemcontainer.Inventory;
import org.l2jmobius.gameserver.model.item.Armor;
import org.l2jmobius.gameserver.model.item.ItemTemplate;
import org.l2jmobius.gameserver.model.item.Weapon;
import org.l2jmobius.gameserver.model.item.enums.BodyPart;
import org.l2jmobius.gameserver.model.item.enums.ItemProcessType;
import org.l2jmobius.gameserver.model.item.holders.ItemHolder;
import org.l2jmobius.gameserver.model.item.instance.Item;
import org.l2jmobius.gameserver.model.item.type.ArmorType;
import org.l2jmobius.gameserver.model.item.type.CrystalType;
import org.l2jmobius.gameserver.model.item.type.WeaponType;
import org.l2jmobius.gameserver.model.skill.holders.SkillHolder;
import org.l2jmobius.gameserver.network.GameClient;
import org.l2jmobius.gameserver.network.clientpackets.CharacterCreate;
import org.l2jmobius.gameserver.network.serverpackets.ExAutoSoulShot;
import org.l2jmobius.gameserver.util.RLAgentNameGenerator;

/**
 * Turns the server into a training environment for autonomous combat agents (see {@code config/Custom/RLAgentEnvironment.ini}).
 * <ul>
 * <li>an agent account without characters gets a random character on login ({@link #onAccountAuthenticated});</li>
 * <li>on its first login an agent is boosted to a random level, promoted along a random class path, taught all skills and equipped with a grade-appropriate set for its archetype;</li>
 * <li>shots and consumables are topped up and automatic shot use is enabled on every login and respawn;</li>
 * <li>agents are teleported to a farm area of their level (spread over the monster spawns, see {@link RLFarmAreas}) and revived there when they die.</li>
 * </ul>
 * Nothing is registered when {@code EnableRLAgentAutoSetup} is off, so the server then behaves as a standard L2J server.
 */
public class RLAgentManager
{
	private static final Logger LOGGER = Logger.getLogger(RLAgentManager.class.getName());
	
	// Player variables.
	/** Set on characters created while the system is enabled: they are agents even if {@code RLApplyToExistingCharacters} is off. */
	private static final String VAR_AGENT = "RL_AGENT";
	/** Set once the first-login setup (level boost) has run. */
	private static final String VAR_SETUP_DONE = "RL_SETUP_DONE";
	/** Grade of the equipment set given last, so a level-up only re-equips when the grade changes. */
	private static final String VAR_GEAR_GRADE = "RL_GEAR_GRADE";
	/** Archetype of the equipment set given last, so a changed weapon rule re-equips existing agents. */
	private static final String VAR_GEAR_ARCHETYPE = "RL_GEAR_ARCHETYPE";
	/** Build of a support class (see {@link Build}). */
	private static final String VAR_BUILD = "RL_BUILD";
	/** The level an agent was created with; a level-up resets it to this level (RLResetLevelOnLevelUp). */
	private static final String VAR_BASE_LEVEL = "RL_BASE_LEVEL";
	/** The class an agent is created for (RLClassBalance): its level is drawn within that profession and its transfers lead to it. */
	private static final String VAR_TARGET_CLASS = "RL_TARGET_CLASS";
	/** How often a summoner agent without a servitor gets one (RLAutoSummon), in ms. */
	private static final long AUTO_SUMMON_INTERVAL = 30000;
	/** Level range of each profession (0 = starting class .. 3 = third profession) for RLClassBalance. */
	private static final int[][] PROFESSION_LEVELS = { { 1, 19 }, { 20, 39 }, { 40, 75 }, { 76, 80 } };
	
	/**
	 * How a support class plays solo. Prophets, Elders, Bishops and their Oracle / Cleric base classes are strong casters only against undead, so they either hunt undead with spells or fight other monsters in melee; Warcryers and Overlords either cast (drains, poisons) or fight with a blunt in heavy armor.
	 */
	enum Build
	{
		/** Magic weapon and robe, mage buffs, only farm areas of undead monsters. */
		UNDEAD_MAGE(true),
		/** Magic weapon and robe, mage buffs, any monsters. */
		CASTER(true),
		/** One-handed blunt, shield and heavy armor, fighter buffs. */
		MELEE_HEAVY(false),
		/** One-handed blunt, shield and light armor, fighter buffs. */
		MELEE_LIGHT(false);
		
		final boolean _mage;
		
		Build(boolean mage)
		{
			_mage = mage;
		}
	}
	
	// Shots indexed by CrystalType ordinal (NONE, D, C, B, A, S).
	private static final int[] SOULSHOTS = { 1835, 1463, 1464, 1465, 1466, 1467 };
	private static final int[] SPIRITSHOTS = { 2509, 2510, 2511, 2512, 2513, 2514 };
	private static final int[] BLESSED_SPIRITSHOTS = { 3947, 3948, 3949, 3950, 3951, 3952 };
	// Arrows of the same grades: Wooden, Bone, Fine Steel, Silver, Mithril, Shining. A bow fires only arrows of its grade.
	private static final int[] ARROWS = { 17, 1341, 1342, 1343, 1344, 1345 };
	// Beast Soulshot and Blessed Beast Spiritshot: the shots of a servitor (it uses its owner's).
	private static final int[] BEAST_SHOTS = { 6645, 6647 };
	// Beast Spiritshot too: an auto-used beast shot is never switched off for the agent's own shots.
	private static final Set<Integer> ALL_BEAST_SHOTS = Set.of(6645, 6646, 6647);
	
	// Levels of the 1st, 2nd and 3rd class transfer.
	private static final int[] CLASS_TRANSFER_LEVELS = { 20, 40, 76 };
	
	// Delays (server timer delays, scaled by TimeScale like every timer).
	/** Lets EnterWorld finish before the character is moved around. */
	private static final long LOGIN_SETUP_DELAY = 2000;
	/** Lets the level change finish (the event is sent before the level is applied). */
	private static final long LEVEL_CHANGE_DELAY = 500;
	/** Completes a teleport whose client never answered with Appearing (headless clients). */
	private static final long TELEPORT_FALLBACK_DELAY = 5000;
	
	/** Candidates of one slot tried before the slot is left empty. */
	private static final int MAX_EQUIP_ATTEMPTS = 10;
	
	private static final int[] GEAR_SLOTS = { Inventory.PAPERDOLL_RHAND, Inventory.PAPERDOLL_LHAND, Inventory.PAPERDOLL_HEAD, Inventory.PAPERDOLL_GLOVES, Inventory.PAPERDOLL_CHEST, Inventory.PAPERDOLL_LEGS, Inventory.PAPERDOLL_FEET, Inventory.PAPERDOLL_NECK, Inventory.PAPERDOLL_REAR, Inventory.PAPERDOLL_LEAR, Inventory.PAPERDOLL_RFINGER, Inventory.PAPERDOLL_LFINGER };
	/** Raid boss jewelry is far above the gear of its grade: never part of an agent's set. */
	private static final String[] BOSS_JEWELRY = { "Queen Ant", "Core", "Orfen", "Antharas", "Valakas", "Baium", "Zaken", "Frintezza" };
	/** Bumped when the equipment rules change, so existing agents get a new set once (2: jewelry). */
	private static final String GEAR_VERSION = "/2";
	
	/** Equipment style of a class. */
	private enum Archetype
	{
		MAGE(ArmorType.MAGIC, false),
		ARCHER(ArmorType.LIGHT, false),
		DAGGER(ArmorType.LIGHT, false),
		MONK(ArmorType.LIGHT, false),
		/** Gladiator, Bladedancer: dual swords only. */
		DUALIST(ArmorType.HEAVY, false),
		/** Warlord: polearms. */
		POLEARM(ArmorType.HEAVY, false),
		/** Destroyer: two-handed swords and blunts. */
		TWO_HANDED(ArmorType.HEAVY, false),
		/** Warrior before the 2nd class transfer: dual swords or polearms (Gladiator or Warlord ahead). */
		WARRIOR(ArmorType.HEAVY, false),
		/** Dwarves: blunts, which several of their combat skills (e.g. Stun Attack, Hammer Crush) require. */
		DWARF(ArmorType.HEAVY, true),
		/** Support classes in melee: one-handed blunt and shield in heavy armor ... */
		SUPPORT_HEAVY(ArmorType.HEAVY, true),
		/** ... or in light armor. */
		SUPPORT_LIGHT(ArmorType.LIGHT, true),
		FIGHTER(ArmorType.HEAVY, true);
		
		final ArmorType _armorType;
		final boolean _usesShield;
		
		Archetype(ArmorType armorType, boolean usesShield)
		{
			_armorType = armorType;
			_usesShield = usesShield;
		}
	}
	
	/** Equipment candidates of one grade and archetype, per slot. */
	private record GearPool(List<ItemTemplate> weapons, List<ItemTemplate> shields, List<ItemTemplate> helmets, List<ItemTemplate> gloves, List<ItemTemplate> boots, List<ItemTemplate> bodies, List<ItemTemplate> legs, List<ItemTemplate> necklaces, List<ItemTemplate> earrings, List<ItemTemplate> rings)
	{
	}
	
	private final Map<CrystalType, Map<Archetype, GearPool>> _gearPools = new ConcurrentHashMap<>();
	/** Players whose setup is running; their own level-up events are ignored meanwhile. */
	private final Set<Integer> _inSetup = ConcurrentHashMap.newKeySet();
	/** Agents per target class (RLClassBalance), read from the database on the first creation. */
	private Map<Integer, Integer> _targetCounts;
	private final Map<Integer, ScheduledFuture<?>> _respawnTasks = new ConcurrentHashMap<>();
	private final RLFarmAreas _farmAreas = new RLFarmAreas();
	private final RLBuffs _buffs = new RLBuffs();
	
	// Listeners
	private final ListenersContainer _listenerContainer = Containers.Players();
	
	private final Consumer<OnPlayerCreate> _playerCreateEvent = event ->
	{
		final Player player = event.getPlayer();
		if (isAgentAccount(player.getAccountName()))
		{
			player.getVariables().set(VAR_AGENT, true);
			player.getVariables().storeMe();
		}
	};
	
	private final Consumer<OnPlayerLogin> _playerLoginEvent = event ->
	{
		final Player player = event.getPlayer();
		if (isAgent(player))
		{
			ThreadPool.schedule(() -> runSafely(player, "login setup", () -> onAgentLogin(player)), LOGIN_SETUP_DELAY);
		}
	};
	
	private final Consumer<OnPlayerLogout> _playerLogoutEvent = event ->
	{
		_farmAreas.release(event.getPlayer());
		_buffs.release(event.getPlayer());
		final ScheduledFuture<?> task = _respawnTasks.remove(event.getPlayer().getObjectId());
		if (task != null)
		{
			task.cancel(false);
		}
	};
	
	private final Consumer<OnPlayerLevelChanged> _playerLevelChangedEvent = event ->
	{
		final Player player = event.getPlayer();
		if ((event.getNewLevel() > event.getOldLevel()) && isAgent(player) && !_inSetup.contains(player.getObjectId()))
		{
			final int baseLevel = player.getVariables().getInt(VAR_BASE_LEVEL, 0);
			if (RLAgentEnvironmentConfig.RL_RESET_LEVEL_ON_LEVEL_UP && (baseLevel > 0) && (event.getNewLevel() > baseLevel))
			{
				// The level population stays as it was drawn: back to the base level, and on to another farm area.
				ThreadPool.schedule(() -> runSafely(player, "level reset", () -> resetToBaseLevel(player, baseLevel)), LEVEL_CHANGE_DELAY);
				return;
			}
			ThreadPool.schedule(() -> runSafely(player, "level-up progression", () ->
			{
				progress(player, false);
				_buffs.relevel(player);
				// A level-up can lift the agent out of its area's level window.
				if (RLAgentEnvironmentConfig.RL_TELEPORT_TO_FARM_ZONE && RLAgentEnvironmentConfig.RL_FARM_AUTO && !player.isDead() && _farmAreas.outgrown(player))
				{
					teleportToFarmZone(player);
				}
			}), LEVEL_CHANGE_DELAY);
		}
	};
	
	private final Consumer<OnCreatureDeath> _creatureDeathEvent = event ->
	{
		if (!RLAgentEnvironmentConfig.RL_RESPAWN_ON_DEATH || !event.getTarget().isPlayer())
		{
			return;
		}
		
		final Player player = event.getTarget().asPlayer();
		if (isAgent(player))
		{
			_respawnTasks.put(player.getObjectId(), ThreadPool.schedule(() -> runSafely(player, "respawn", () -> respawn(player)), RLAgentEnvironmentConfig.RL_RESPAWN_DELAY));
		}
	};
	
	protected RLAgentManager()
	{
		_listenerContainer.addListener(new ConsumerEventListener(_listenerContainer, EventType.ON_PLAYER_CREATE, _playerCreateEvent, this));
		_listenerContainer.addListener(new ConsumerEventListener(_listenerContainer, EventType.ON_PLAYER_LOGIN, _playerLoginEvent, this));
		_listenerContainer.addListener(new ConsumerEventListener(_listenerContainer, EventType.ON_PLAYER_LOGOUT, _playerLogoutEvent, this));
		_listenerContainer.addListener(new ConsumerEventListener(_listenerContainer, EventType.ON_PLAYER_LEVEL_CHANGED, _playerLevelChangedEvent, this));
		_listenerContainer.addListener(new ConsumerEventListener(_listenerContainer, EventType.ON_CREATURE_DEATH, _creatureDeathEvent, this));
		if (RLAgentEnvironmentConfig.RL_AUTO_SUMMON)
		{
			ThreadPool.scheduleAtFixedRate(this::ensureSummons, AUTO_SUMMON_INTERVAL, AUTO_SUMMON_INTERVAL);
		}
		LOGGER.info(getClass().getSimpleName() + ": Agent levels " + RLAgentEnvironmentConfig.RL_MIN_LEVEL + "-" + RLAgentEnvironmentConfig.RL_MAX_LEVEL + ", accounts matching \"" + RLAgentEnvironmentConfig.RL_AGENT_ACCOUNT_PATTERN.pattern() + "\", " + RLAgentEnvironmentConfig.RL_FARM_ZONES.size() + " farm zone bracket(s).");
	}
	
	// --------------------------------------------------------------------------------
	// Agent detection
	// --------------------------------------------------------------------------------
	
	private static boolean isAgentAccount(String accountName)
	{
		return RLAgentEnvironmentConfig.RL_AGENT_ENABLED && (accountName != null) && RLAgentEnvironmentConfig.RL_AGENT_ACCOUNT_PATTERN.matcher(accountName).matches();
	}
	
	/**
	 * Casts the full buffer set ({@code RLBuffFull}) on a player and its summon, for any player: the {@code .fullbuff} command.
	 * @param player the player
	 * @return the number of buffs cast
	 */
	public int giveFullBuff(Player player)
	{
		return RLBuffs.giveFull(player);
	}
	
	/**
	 * @param player the player to check
	 * @return {@code true} if the automated environment manages this character
	 */
	public boolean isAgent(Player player)
	{
		return (player != null) && isAgentAccount(player.getAccountName()) && (RLAgentEnvironmentConfig.RL_APPLY_TO_EXISTING_CHARACTERS || player.getVariables().getBoolean(VAR_AGENT, false));
	}
	
	private void runSafely(Player player, String what, Runnable action)
	{
		try
		{
			action.run();
		}
		catch (Exception e)
		{
			LOGGER.log(Level.WARNING, getClass().getSimpleName() + ": " + what + " failed for " + player + ": " + e, e);
		}
	}
	
	// --------------------------------------------------------------------------------
	// Character creation
	// --------------------------------------------------------------------------------
	
	/**
	 * Called right before the character list is sent to a freshly authenticated client: an agent account without characters gets a random one, so a headless client can select slot 0 immediately.
	 * @param client the authenticated client
	 */
	public void onAccountAuthenticated(GameClient client)
	{
		final String account = client.getAccountName();
		if (!RLAgentEnvironmentConfig.RL_AUTO_CREATE_CHARACTER || !isAgentAccount(account))
		{
			return;
		}
		
		final List<PlayerClass> baseClasses = new ArrayList<>();
		for (PlayerClass playerClass : PlayerClass.values())
		{
			if ((playerClass.level() == 0) && isRaceAllowed(playerClass) && (PlayerTemplateData.getInstance().getTemplate(playerClass) != null))
			{
				baseClasses.add(playerClass);
			}
		}
		if (baseClasses.isEmpty())
		{
			LOGGER.warning(getClass().getSimpleName() + ": No allowed base class, cannot create a character for " + account + ".");
			return;
		}
		
		final PlayerClass target = RLAgentEnvironmentConfig.RL_CLASS_BALANCE ? pickTargetClass(baseClasses) : null;
		final PlayerClass baseClass = target != null ? target.getRootClass() : baseClasses.get(Rnd.get(baseClasses.size()));
		final PlayerTemplate template = PlayerTemplateData.getInstance().getTemplate(baseClass);
		final boolean female = Rnd.nextBoolean();
		final PlayerAppearance appearance = new PlayerAppearance((byte) Rnd.get(0, 2), (byte) Rnd.get(0, 3), (byte) Rnd.get(0, female ? 6 : 4), female);
		
		final Player newChar;
		synchronized (CharInfoTable.getInstance())
		{
			if (CharInfoTable.getInstance().getAccountCharacterCount(account) > 0)
			{
				return;
			}
			
			final String name = RLAgentNameGenerator.generate();
			if (name == null)
			{
				LOGGER.warning(getClass().getSimpleName() + ": Could not generate a free character name for " + account + ".");
				return;
			}
			
			newChar = Player.create(template, account, name, appearance);
		}
		if (newChar == null)
		{
			return;
		}
		
		newChar.setCurrentHp(newChar.getMaxHp());
		newChar.setCurrentMp(newChar.getMaxMp());
		CharacterCreate.initNewChar(client, newChar);
		if (target != null)
		{
			newChar.getVariables().set(VAR_TARGET_CLASS, target.getId());
			newChar.getVariables().storeMe();
		}
		LOGGER.info(getClass().getSimpleName() + ": Created agent character " + newChar.getName() + " (" + baseClass + (target != null ? ", for " + target : "") + ") for account " + account + ".");
	}
	
	/**
	 * The class with the fewest agents (ties at random) among every class whose starting class may be created: the agents spread evenly over all professions.
	 * @param baseClasses the starting classes that may be created
	 * @return the class the new agent is created for
	 */
	private synchronized PlayerClass pickTargetClass(List<PlayerClass> baseClasses)
	{
		if (_targetCounts == null)
		{
			_targetCounts = new HashMap<>();
			try (Connection con = DatabaseFactory.getConnection();
				PreparedStatement ps = con.prepareStatement("SELECT val, COUNT(*) FROM character_variables WHERE var=? GROUP BY val");)
			{
				ps.setString(1, VAR_TARGET_CLASS);
				try (ResultSet rs = ps.executeQuery())
				{
					while (rs.next())
					{
						_targetCounts.put(Integer.parseInt(rs.getString(1)), rs.getInt(2));
					}
				}
			}
			catch (Exception e)
			{
				LOGGER.warning(getClass().getSimpleName() + ": Could not count the agents per class: " + e);
			}
		}
		
		final List<PlayerClass> fewest = new ArrayList<>();
		double lowest = Double.MAX_VALUE;
		for (PlayerClass playerClass : PlayerClass.values())
		{
			if ((playerClass.level() >= PROFESSION_LEVELS.length) || !baseClasses.contains(playerClass.getRootClass()) || !professionReachable(playerClass.level()))
			{
				continue;
			}
			// Agents per unit of the profession's weight: a profession with weight 0.5 gets half the agents per class.
			final double weight = RLAgentEnvironmentConfig.RL_CLASS_BALANCE_WEIGHTS[Math.min(playerClass.level(), RLAgentEnvironmentConfig.RL_CLASS_BALANCE_WEIGHTS.length - 1)];
			if (weight <= 0)
			{
				continue;
			}
			final double load = _targetCounts.getOrDefault(playerClass.getId(), 0) / weight;
			if (load < (lowest - 1e-9))
			{
				lowest = load;
				fewest.clear();
			}
			if (Math.abs(load - lowest) < 1e-9)
			{
				fewest.add(playerClass);
			}
		}
		if (fewest.isEmpty())
		{
			return null;
		}
		final PlayerClass target = fewest.get(Rnd.get(fewest.size()));
		_targetCounts.merge(target.getId(), 1, Integer::sum);
		return target;
	}
	
	/**
	 * @param profession 0 (starting class) .. 3 (third profession)
	 * @return {@code true} when the profession's level range overlaps [RLMinLevel, RLMaxLevel]
	 */
	private static boolean professionReachable(int profession)
	{
		return (PROFESSION_LEVELS[profession][0] <= RLAgentEnvironmentConfig.RL_MAX_LEVEL) && (PROFESSION_LEVELS[profession][1] >= RLAgentEnvironmentConfig.RL_MIN_LEVEL);
	}
	
	private static boolean isRaceAllowed(PlayerClass playerClass)
	{
		return switch (playerClass.getRace())
		{
			case HUMAN -> AllowedPlayerRacesConfig.ALLOW_HUMAN;
			case ELF -> AllowedPlayerRacesConfig.ALLOW_ELF;
			case DARK_ELF -> AllowedPlayerRacesConfig.ALLOW_DARKELF;
			case ORC -> AllowedPlayerRacesConfig.ALLOW_ORC;
			case DWARF -> AllowedPlayerRacesConfig.ALLOW_DWARF;
			default -> false;
		};
	}
	
	// --------------------------------------------------------------------------------
	// Login setup, progression and respawn
	// --------------------------------------------------------------------------------
	
	private void onAgentLogin(Player player)
	{
		if (!player.isOnline())
		{
			return;
		}
		
		final boolean firstSetup = !player.getVariables().getBoolean(VAR_SETUP_DONE, false);
		_inSetup.add(player.getObjectId());
		try
		{
			if (firstSetup)
			{
				boostLevel(player);
			}
			
			progress(player, firstSetup);
			
			if (player.isDead())
			{
				player.doRevive();
			}
			restoreVitals(player);
			
			if (RLAgentEnvironmentConfig.RL_TELEPORT_TO_FARM_ZONE && (firstSetup || RLAgentEnvironmentConfig.RL_TELEPORT_ON_LOGIN))
			{
				teleportToFarmZone(player);
			}
			// The server restores the buffs of the last session on login: a life starts with exactly one buff set.
			player.stopAllEffectsExceptThoseThatLastThroughDeath();
			clearDeathPenalty(player);
			_buffs.assign(player);
			ensureSummon(player);
			
			if (firstSetup)
			{
				player.getVariables().set(VAR_SETUP_DONE, true);
				LOGGER.info(getClass().getSimpleName() + ": Agent " + player.getName() + " set up: level " + player.getLevel() + ", " + player.getPlayerClass() + ".");
			}
			player.getVariables().storeMe();
			player.store(false);
		}
		finally
		{
			_inSetup.remove(player.getObjectId());
		}
	}
	
	/** Takes an agent that levelled up back to the start of its base level and moves it to another farm area. */
	private void resetToBaseLevel(Player player, int baseLevel)
	{
		final long baseExp = ExperienceData.getInstance().getExpForLevel(baseLevel);
		if (!player.isOnline() || (player.getExp() <= baseExp))
		{
			return;
		}
		player.removeExpAndSp(player.getExp() - baseExp, 0, false);
		if (RLAgentEnvironmentConfig.RL_TELEPORT_TO_FARM_ZONE && RLAgentEnvironmentConfig.RL_FARM_AUTO && !player.isDead())
		{
			final Location spot = _farmAreas.relocate(player);
			if (spot != null)
			{
				teleportTo(player, spot);
			}
		}
	}
	
	private void boostLevel(Player player)
	{
		final int maxLevel = Math.min(RLAgentEnvironmentConfig.RL_MAX_LEVEL, ExperienceData.getInstance().getMaxLevel() - 1);
		final int minLevel = Math.min(RLAgentEnvironmentConfig.RL_MIN_LEVEL, maxLevel);
		final int targetLevel;
		final int targetClass = player.getVariables().getInt(VAR_TARGET_CLASS, -1);
		if (RLAgentEnvironmentConfig.RL_CLASS_BALANCE && (targetClass >= 0) && (PlayerClass.getPlayerClass(targetClass) != null))
		{
			// Uniformly within the levels of the target class's profession (and the configured range).
			final int[] range = PROFESSION_LEVELS[Math.min(PlayerClass.getPlayerClass(targetClass).level(), PROFESSION_LEVELS.length - 1)];
			final int low = Math.max(range[0], minLevel);
			final int high = Math.max(low, Math.min(range[1], maxLevel));
			targetLevel = Rnd.get(low, high);
		}
		else if (RLAgentEnvironmentConfig.RL_LEVEL_STDDEV > 0)
		{
			// Gaussian around RLLevelMean, redrawn until it falls into the level range (a truncated normal distribution).
			int level;
			do
			{
				level = (int) Math.round(RLAgentEnvironmentConfig.RL_LEVEL_MEAN + (Rnd.nextGaussian() * RLAgentEnvironmentConfig.RL_LEVEL_STDDEV));
			}
			while ((level < minLevel) || (level > maxLevel));
			targetLevel = level;
		}
		else
		{
			targetLevel = Rnd.get(minLevel, maxLevel);
		}
		// The level the agent is created with: its level-ups are reset to it (RLResetLevelOnLevelUp).
		player.getVariables().set(VAR_BASE_LEVEL, Math.max(targetLevel, player.getLevel()));
		if (player.getLevel() >= targetLevel)
		{
			return;
		}
		
		player.addExpAndSp(ExperienceData.getInstance().getExpForLevel(targetLevel) - player.getExp(), 0);
	}
	
	/**
	 * Applies everything that depends on the level: class transfers, skills, equipment and supplies.
	 * @param player the agent
	 * @param forceEquip {@code true} to give a new equipment set even if the grade did not change
	 */
	private void progress(Player player, boolean forceEquip)
	{
		if (!player.isOnline())
		{
			return;
		}
		
		if (RLAgentEnvironmentConfig.RL_AUTO_CLASS_TRANSFER)
		{
			transferClasses(player);
		}
		
		if (RLAgentEnvironmentConfig.RL_AUTO_LEARN_SKILLS)
		{
			learnSkills(player);
		}
		
		assignBuild(player);
		
		if (RLAgentEnvironmentConfig.RL_AUTO_EQUIP)
		{
			final CrystalType grade = gradeForLevel(player.getLevel());
			final String archetype = archetypeFor(player).name() + GEAR_VERSION;
			if (forceEquip || !grade.name().equals(player.getVariables().getString(VAR_GEAR_GRADE, "")) || !archetype.equals(player.getVariables().getString(VAR_GEAR_ARCHETYPE, "")))
			{
				equip(player, grade);
			}
		}
		
		supply(player);
	}
	
	private void respawn(Player player)
	{
		_respawnTasks.remove(player.getObjectId());
		if (!player.isOnline() || !player.isDead())
		{
			return;
		}
		
		player.doRevive();
		clearDeathPenalty(player);
		restoreVitals(player);
		teleportToFarmZone(player);
		supply(player);
		_buffs.assign(player); // a new life, a new buff set
		ensureSummon(player);
	}
	
	/** Removes the Death Penalty debuff an agent still carries (from before RLAgentDeathPenalty = False, or from a PvP death). */
	private static void clearDeathPenalty(Player player)
	{
		if (RLAgentEnvironmentConfig.RL_DEATH_PENALTY)
		{
			return;
		}
		while (player.getDeathPenaltyBuffLevel() > 0)
		{
			player.reduceDeathPenaltyBuffLevel();
		}
	}
	
	private static void restoreVitals(Player player)
	{
		player.setCurrentHpMp(player.getMaxHp(), player.getMaxMp());
		player.setCurrentCp(player.getMaxCp());
	}
	
	// --------------------------------------------------------------------------------
	// Class transfer and skills
	// --------------------------------------------------------------------------------
	
	private void transferClasses(Player player)
	{
		boolean changed = false;
		while (true)
		{
			final PlayerClass current = player.getPlayerClass();
			final int transfer = current.level();
			if ((transfer >= CLASS_TRANSFER_LEVELS.length) || (player.getLevel() < CLASS_TRANSFER_LEVELS[transfer]))
			{
				break;
			}
			
			final List<PlayerClass> options = new ArrayList<>(current.getNextClasses());
			if (options.isEmpty())
			{
				break;
			}
			
			// An agent created for a class (RLClassBalance) goes the way to it; any other one takes a random branch.
			final PlayerClass target = PlayerClass.getPlayerClass(player.getVariables().getInt(VAR_TARGET_CLASS, -1));
			PlayerClass next = null;
			if (target != null)
			{
				for (PlayerClass option : options)
				{
					if (target.equalsOrChildOf(option))
					{
						next = option;
						break;
					}
				}
			}
			if (next == null)
			{
				next = options.get(Rnd.get(options.size()));
			}
			player.setPlayerClass(next.getId());
			if (player.isSubClassActive())
			{
				player.getSubClasses().get(player.getClassIndex()).setPlayerClass(player.getActiveClass());
			}
			else
			{
				player.setBaseClass(player.getActiveClass());
			}
			changed = true;
		}
		
		if (changed)
		{
			player.broadcastUserInfo();
		}
	}
	
	private static void learnSkills(Player player)
	{
		// A class transfer can unlock skills that in turn are prerequisites, so repeat until nothing new is learned.
		for (int round = 0; (round < 10) && (player.giveAvailableSkills(false, true, true) > 0); round++)
		{
			// Keep learning.
		}
		player.sendSkillList();
	}
	
	// --------------------------------------------------------------------------------
	// Equipment
	// --------------------------------------------------------------------------------
	
	private static CrystalType gradeForLevel(int level)
	{
		CrystalType grade = CrystalType.NONE;
		for (Entry<Integer, CrystalType> entry : RLAgentEnvironmentConfig.RL_GRADE_LEVELS.entrySet())
		{
			if (level >= entry.getKey())
			{
				grade = entry.getValue();
			}
		}
		return grade;
	}
	
	/** The equipment archetype of an agent: its build's for a support class, its class's otherwise. */
	private static Archetype archetypeFor(Player player)
	{
		final Build build = buildOf(player);
		if (build != null)
		{
			return switch (build)
			{
				case UNDEAD_MAGE, CASTER -> Archetype.MAGE;
				case MELEE_HEAVY -> Archetype.SUPPORT_HEAVY;
				case MELEE_LIGHT -> Archetype.SUPPORT_LIGHT;
			};
		}
		return archetypeOf(player.getPlayerClass());
	}
	
	/** Class family of the builds ({@code RLBuilds.<family>}), {@code null} for a class without builds. */
	private static String buildFamily(PlayerClass playerClass)
	{
		if (playerClass.equalsOrChildOf(PlayerClass.PROPHET))
		{
			return "Prophet";
		}
		if (playerClass.equalsOrChildOf(PlayerClass.CLERIC) || playerClass.equalsOrChildOf(PlayerClass.ORACLE) || playerClass.equalsOrChildOf(PlayerClass.SHILLIEN_ORACLE))
		{
			return "Elder";
		}
		if (playerClass.equalsOrChildOf(PlayerClass.ORC_SHAMAN))
		{
			return "Shaman";
		}
		return null;
	}
	
	/** The agent's build, {@code null} for a class without builds. */
	static Build buildOf(Player player)
	{
		if (buildFamily(player.getPlayerClass()) == null)
		{
			return null;
		}
		try
		{
			return Build.valueOf(player.getVariables().getString(VAR_BUILD, ""));
		}
		catch (IllegalArgumentException e)
		{
			return null;
		}
	}
	
	/** {@code true} when the agent plays as a caster: its build says so, or (without builds) its class is a mage. */
	static boolean isMageBuild(Player player)
	{
		final Build build = buildOf(player);
		return (build != null) ? build._mage : player.isMageClass();
	}
	
	/**
	 * Picks a build for a support class, or a new one when the current build no longer fits (a class transfer to another family, undead hunting below {@code RLUndeadMinLevel}).
	 */
	private static void assignBuild(Player player)
	{
		final String family = buildFamily(player.getPlayerClass());
		if (family == null)
		{
			return;
		}
		
		final Map<String, Integer> weights = RLAgentEnvironmentConfig.RL_BUILDS.getOrDefault(family, Map.of());
		final Build current = buildOf(player);
		if ((current != null) && weights.containsKey(current.name().toLowerCase()) && ((current != Build.UNDEAD_MAGE) || (player.getLevel() >= RLAgentEnvironmentConfig.RL_UNDEAD_MIN_LEVEL)))
		{
			return;
		}
		
		final List<Build> builds = new ArrayList<>();
		final List<Integer> buildWeights = new ArrayList<>();
		int total = 0;
		for (Entry<String, Integer> entry : weights.entrySet())
		{
			final Build build;
			try
			{
				build = Build.valueOf(entry.getKey().toUpperCase());
			}
			catch (IllegalArgumentException e)
			{
				continue;
			}
			if ((entry.getValue() <= 0) || ((build == Build.UNDEAD_MAGE) && (player.getLevel() < RLAgentEnvironmentConfig.RL_UNDEAD_MIN_LEVEL)))
			{
				continue;
			}
			builds.add(build);
			buildWeights.add(entry.getValue());
			total += entry.getValue();
		}
		if (builds.isEmpty())
		{
			player.getVariables().remove(VAR_BUILD);
			return;
		}
		
		int pick = Rnd.get(total);
		Build chosen = builds.get(builds.size() - 1);
		for (int i = 0; i < builds.size(); i++)
		{
			pick -= buildWeights.get(i);
			if (pick < 0)
			{
				chosen = builds.get(i);
				break;
			}
		}
		player.getVariables().set(VAR_BUILD, chosen.name());
	}
	
	private static Archetype archetypeOf(PlayerClass playerClass)
	{
		if (playerClass.isMage())
		{
			return Archetype.MAGE;
		}
		if (playerClass.equalsOrChildOf(PlayerClass.HAWKEYE) || playerClass.equalsOrChildOf(PlayerClass.SILVER_RANGER) || playerClass.equalsOrChildOf(PlayerClass.PHANTOM_RANGER))
		{
			return Archetype.ARCHER;
		}
		if (playerClass.equalsOrChildOf(PlayerClass.ROGUE) || playerClass.equalsOrChildOf(PlayerClass.ELVEN_SCOUT) || playerClass.equalsOrChildOf(PlayerClass.ASSASSIN))
		{
			return Archetype.DAGGER;
		}
		if (playerClass.equalsOrChildOf(PlayerClass.ORC_MONK))
		{
			return Archetype.MONK;
		}
		if (playerClass.equalsOrChildOf(PlayerClass.GLADIATOR) || playerClass.equalsOrChildOf(PlayerClass.BLADEDANCER))
		{
			return Archetype.DUALIST;
		}
		if (playerClass.equalsOrChildOf(PlayerClass.WARLORD))
		{
			return Archetype.POLEARM;
		}
		if (playerClass.equalsOrChildOf(PlayerClass.ORC_RAIDER))
		{
			return Archetype.TWO_HANDED;
		}
		if (playerClass == PlayerClass.WARRIOR)
		{
			return Archetype.WARRIOR;
		}
		if (playerClass.equalsOrChildOf(PlayerClass.DWARVEN_FIGHTER))
		{
			return Archetype.DWARF;
		}
		return Archetype.FIGHTER;
	}
	
	private void equip(Player player, CrystalType grade)
	{
		final GearPool pool = _gearPools.computeIfAbsent(grade, g -> new EnumMap<>(Archetype.class)).computeIfAbsent(archetypeFor(player), archetype -> buildGearPool(grade, archetype));
		if (pool.weapons().isEmpty())
		{
			LOGGER.warning(getClass().getSimpleName() + ": No " + grade + " weapon for " + player.getPlayerClass() + ".");
			return;
		}
		
		// Agent inventories are disposable: the replaced set is destroyed instead of piling up.
		final Inventory inventory = player.getInventory();
		final Set<Integer> removed = new HashSet<>();
		for (int slot : GEAR_SLOTS)
		{
			final Item old = inventory.getPaperdollItem(slot);
			if ((old != null) && removed.add(old.getObjectId()))
			{
				inventory.unEquipItemInSlot(slot);
				player.destroyItem(ItemProcessType.NONE, old, null, false);
			}
		}
		
		final Item weapon = equipFirstFitting(player, pool.weapons());
		if ((weapon == null) || (weapon.getTemplate().getBodyPart() != BodyPart.LR_HAND))
		{
			equipFirstFitting(player, pool.shields());
		}
		equipFirstFitting(player, pool.helmets());
		equipFirstFitting(player, pool.gloves());
		equipFirstFitting(player, pool.boots());
		final Item body = equipFirstFitting(player, pool.bodies());
		if ((body == null) || (body.getTemplate().getBodyPart() != BodyPart.FULL_ARMOR))
		{
			equipFirstFitting(player, pool.legs());
		}
		// Jewelry: a necklace, two earrings and two rings (the second of a pair goes to the free slot).
		equipFirstFitting(player, pool.necklaces());
		equipFirstFitting(player, pool.earrings());
		equipFirstFitting(player, pool.earrings());
		equipFirstFitting(player, pool.rings());
		equipFirstFitting(player, pool.rings());
		
		player.getVariables().set(VAR_GEAR_GRADE, grade.name());
		player.getVariables().set(VAR_GEAR_ARCHETYPE, archetypeFor(player).name() + GEAR_VERSION);
		player.sendItemList(false);
		player.broadcastUserInfo();
	}
	
	/**
	 * Equips a random candidate the player can actually wear: items with an equip condition the agent does not meet (e.g. clan-rank Apella armor) are skipped, and an item the inventory refuses to equip is destroyed and the next one is tried.
	 * @param player the agent
	 * @param candidates the slot's candidates
	 * @return the equipped item, or {@code null} if the slot stays empty
	 */
	private static Item equipFirstFitting(Player player, List<ItemTemplate> candidates)
	{
		final List<ItemTemplate> shuffled = new ArrayList<>(candidates);
		Collections.shuffle(shuffled);
		int attempts = 0;
		for (ItemTemplate template : shuffled)
		{
			if (attempts >= MAX_EQUIP_ATTEMPTS)
			{
				break;
			}
			if (!template.checkCondition(player, player, false))
			{
				continue;
			}
			
			attempts++;
			final Item item = player.getInventory().addItem(ItemProcessType.REWARD, template.getId(), 1, player, null);
			if (item == null)
			{
				continue;
			}
			player.getInventory().equipItem(item);
			if (item.isEquipped())
			{
				return item;
			}
			player.destroyItem(ItemProcessType.NONE, item, null, false);
		}
		return null;
	}
	
	/**
	 * A/S gloves and boots come in armor-type variants ("... - Heavy Armor", "... - Light Armor", "... - Robe") whose set bonus only works with that armor type.
	 * @param item the armor piece
	 * @param archetype the agent's archetype
	 * @return {@code false} for a variant of another armor type
	 */
	private static boolean fitsArmorVariant(ItemTemplate item, Archetype archetype)
	{
		final String name = item.getName();
		if (name.endsWith("- Heavy Armor"))
		{
			return archetype._armorType == ArmorType.HEAVY;
		}
		if (name.endsWith("- Light Armor"))
		{
			return archetype._armorType == ArmorType.LIGHT;
		}
		if (name.endsWith("- Robe"))
		{
			return archetype._armorType == ArmorType.MAGIC;
		}
		return true;
	}
	
	private static GearPool buildGearPool(CrystalType grade, Archetype archetype)
	{
		// Sealed A/S pieces lack the set bonus until unsealed, test and time-limited (shadow) items are not real gear.
		// Items whose skill is missing from the datapack (e.g. the "- Light" weapons with skill 3498) fail on login.
		final Predicate<ItemTemplate> usable = item -> (item.getCrystalType() == grade) && (item.getReferencePrice() > 0) && (item.getDuration() <= 0) && !item.isQuestItem() && !item.isHeroItem() && !item.isPvpItem() && !item.isCommon() && !item.getName().contains("Test") && !item.getName().startsWith("Sealed ") && hasKnownSkills(item);
		
		final List<ItemTemplate> weapons = new ArrayList<>();
		for (Weapon weapon : ItemData.getInstance().getAllWeapons())
		{
			if (usable.test(weapon) && fitsArchetype(weapon, archetype))
			{
				weapons.add(weapon);
			}
		}
		
		final List<ItemTemplate> shields = new ArrayList<>();
		final List<ItemTemplate> helmets = new ArrayList<>();
		final List<ItemTemplate> gloves = new ArrayList<>();
		final List<ItemTemplate> boots = new ArrayList<>();
		final List<ItemTemplate> bodies = new ArrayList<>();
		final List<ItemTemplate> legs = new ArrayList<>();
		final List<ItemTemplate> necklaces = new ArrayList<>();
		final List<ItemTemplate> earrings = new ArrayList<>();
		final List<ItemTemplate> rings = new ArrayList<>();
		for (Armor armor : ItemData.getInstance().getAllArmors())
		{
			if (!usable.test(armor) || !fitsArmorVariant(armor, archetype))
			{
				continue;
			}
			
			final BodyPart bodyPart = armor.getBodyPart();
			final boolean ownType = armor.getItemType() == archetype._armorType;
			if ((bodyPart == BodyPart.L_HAND) && archetype._usesShield)
			{
				shields.add(armor);
			}
			else if (bodyPart == BodyPart.HEAD)
			{
				helmets.add(armor);
			}
			else if (bodyPart == BodyPart.GLOVES)
			{
				gloves.add(armor);
			}
			else if (bodyPart == BodyPart.FEET)
			{
				boots.add(armor);
			}
			else if (((bodyPart == BodyPart.CHEST) || (bodyPart == BodyPart.FULL_ARMOR)) && ownType)
			{
				bodies.add(armor);
			}
			else if ((bodyPart == BodyPart.LEGS) && ownType)
			{
				legs.add(armor);
			}
			else if (((bodyPart == BodyPart.NECK) || (bodyPart == BodyPart.LR_EAR) || (bodyPart == BodyPart.LR_FINGER)) && !isBossJewelry(armor))
			{
				(bodyPart == BodyPart.NECK ? necklaces : bodyPart == BodyPart.LR_EAR ? earrings : rings).add(armor);
			}
		}
		
		LOGGER.info(RLAgentManager.class.getSimpleName() + ": " + grade + " " + archetype + " gear pool: " + weapons.size() + " weapons, " + shields.size() + " shields, " + helmets.size() + " helmets, " + gloves.size() + " gloves, " + boots.size() + " boots, " + bodies.size() + " bodies, " + legs.size() + " legs, " + necklaces.size() + " necklaces, " + earrings.size() + " earrings, " + rings.size() + " rings.");
		return new GearPool(weapons, shields, helmets, gloves, boots, bodies, legs, necklaces, earrings, rings);
	}
	
	/** The skill brings a servitor (the Summon effect handler reports no effect type of its own, so it is found by class). */
	/** Summons a servitor for every agent that can summon one and has none (RLAutoSummon). */
	private void ensureSummons()
	{
		for (Player player : World.getInstance().getPlayers())
		{
			if (isAgent(player))
			{
				runSafely(player, "auto summon", () -> ensureSummon(player));
			}
		}
	}
	
	/**
	 * A summoner agent gets its servitor without the 15 s cast (RLAutoSummon), as a buffer gives its buffs: a cast in a fight is interrupted and the low-level summons have a 6 h reuse, so the agents would hardly ever play with a servitor. One of its summoning skills at random; the crystals are paid as for a cast.
	 * @param player the agent
	 */
	private static void ensureSummon(Player player)
	{
		if (!RLAgentEnvironmentConfig.RL_AUTO_SUMMON || !player.isOnline() || player.isDead() || player.hasSummon())
		{
			return;
		}
		final List<Skill> summons = new ArrayList<>();
		for (Skill skill : player.getAllSkills())
		{
			if (!skill.isPassive() && summonsServitor(skill) && !RLAgentEnvironmentConfig.RL_AUTO_SUMMON_EXCLUDED.contains(skill.getId()))
			{
				summons.add(skill);
			}
		}
		if (summons.isEmpty())
		{
			return;
		}
		final Skill skill = summons.get(Rnd.get(summons.size()));
		if ((skill.getItemConsumeId() > 0) && (skill.getItemConsumeCount() > 0) && !player.destroyItemByItemId(ItemProcessType.NONE, skill.getItemConsumeId(), skill.getItemConsumeCount(), null, false))
		{
			return;
		}
		skill.applyEffects(player, player);
	}
	
	private static boolean summonsServitor(Skill skill)
	{
		final List<AbstractEffect> effects = skill.getEffects(EffectScope.GENERAL);
		if (effects == null)
		{
			return false;
		}
		for (AbstractEffect effect : effects)
		{
			if ((effect != null) && effect.getClass().getSimpleName().equals("Summon"))
			{
				return true;
			}
		}
		return false;
	}
	
	private static boolean isBossJewelry(ItemTemplate item)
	{
		for (String boss : BOSS_JEWELRY)
		{
			if (item.getName().contains(boss))
			{
				return true;
			}
		}
		return false;
	}
	
	private static boolean hasKnownSkills(ItemTemplate item)
	{
		final SkillHolder[] skills = item.getSkills();
		if (skills != null)
		{
			for (SkillHolder holder : skills)
			{
				// Checked via the level table: SkillHolder.getSkill() logs a stack trace for every missing skill.
				if (SkillData.getInstance().getMaxLevel(holder.getSkillId()) < holder.getSkillLevel())
				{
					return false;
				}
			}
		}
		return true;
	}
	
	private static boolean fitsArchetype(Weapon weapon, Archetype archetype)
	{
		final WeaponType type = weapon.getItemType();
		return switch (archetype)
		{
			case MAGE -> weapon.isMagicWeapon() && ((type == WeaponType.SWORD) || (type == WeaponType.BLUNT));
			case ARCHER -> type == WeaponType.BOW;
			case DAGGER -> type == WeaponType.DAGGER;
			case MONK -> (type == WeaponType.DUALFIST) || (type == WeaponType.FIST);
			case DUALIST -> type == WeaponType.DUAL;
			case POLEARM -> type == WeaponType.POLE;
			case TWO_HANDED -> !weapon.isMagicWeapon() && ((type == WeaponType.SWORD) || (type == WeaponType.BLUNT)) && (weapon.getBodyPart() == BodyPart.LR_HAND);
			case WARRIOR -> (type == WeaponType.DUAL) || (type == WeaponType.POLE);
			case DWARF -> !weapon.isMagicWeapon() && (type == WeaponType.BLUNT);
			case SUPPORT_HEAVY, SUPPORT_LIGHT -> !weapon.isMagicWeapon() && (type == WeaponType.BLUNT) && (weapon.getBodyPart() == BodyPart.R_HAND);
			case FIGHTER -> !weapon.isMagicWeapon() && ((type == WeaponType.SWORD) || (type == WeaponType.BLUNT)) && (weapon.getBodyPart() == BodyPart.R_HAND);
		};
	}
	
	// --------------------------------------------------------------------------------
	// Supplies
	// --------------------------------------------------------------------------------
	
	private void supply(Player player)
	{
		final Weapon weapon = player.getActiveWeaponItem();
		final int gradeIndex = (weapon == null) ? 0 : Math.min(weapon.getCrystalType().ordinal(), SOULSHOTS.length - 1);
		final int shotId;
		final int shotCount;
		if (isMageBuild(player))
		{
			shotId = RLAgentEnvironmentConfig.RL_USE_BLESSED_SPIRITSHOTS ? BLESSED_SPIRITSHOTS[gradeIndex] : SPIRITSHOTS[gradeIndex];
			shotCount = RLAgentEnvironmentConfig.RL_SPIRITSHOT_COUNT;
		}
		else
		{
			shotId = SOULSHOTS[gradeIndex];
			shotCount = RLAgentEnvironmentConfig.RL_SOULSHOT_COUNT;
		}
		
		topUp(player, shotId, shotCount);
		// Arrows of the bow's grade only: those of earlier grades (or of a bow no longer carried) would
		// pile up past the weight limit (an arrow weighs 4-6).
		final boolean bow = (weapon != null) && (weapon.getItemType() == WeaponType.BOW);
		for (int arrow : ARROWS)
		{
			if (!bow || (arrow != ARROWS[gradeIndex]))
			{
				final long count = player.getInventory().getInventoryItemCount(arrow, -1);
				if (count > 0)
				{
					player.destroyItemByItemId(ItemProcessType.NONE, arrow, (int) count, null, false);
				}
			}
		}
		if (bow)
		{
			// The server equips matching arrows by itself when the bow attacks.
			topUp(player, ARROWS[gradeIndex], RLAgentEnvironmentConfig.RL_ARROW_COUNT);
		}
		for (ItemHolder consumable : RLAgentEnvironmentConfig.RL_CONSUMABLES)
		{
			topUp(player, consumable.getId(), consumable.getCount());
		}
		// What the agent's skills consume (summoning crystals, spirit ore) and its servitor's shots.
		boolean summoner = false;
		for (Skill skill : player.getAllSkills())
		{
			if (skill.isPassive())
			{
				continue;
			}
			if ((skill.getItemConsumeId() > 0) && (skill.getItemConsumeCount() > 0))
			{
				topUp(player, skill.getItemConsumeId(), Math.max(RLAgentEnvironmentConfig.RL_SKILL_ITEM_COUNT, skill.getItemConsumeCount() * 10));
			}
			summoner |= summonsServitor(skill);
		}
		if (summoner)
		{
			for (int beastShot : BEAST_SHOTS)
			{
				topUp(player, beastShot, RLAgentEnvironmentConfig.RL_BEAST_SHOT_COUNT);
			}
		}
		// Buff potions; the actor drinks one when no buff of the same kind is on the character.
		for (PotionEntry potion : RLAgentEnvironmentConfig.RL_BUFF_POTIONS)
		{
			final int level = player.getLevel();
			if ((level >= potion.minLevel()) && (level <= potion.maxLevel()) && (!potion.mageOnly() || isMageBuild(player)) && (!potion.fighterOnly() || !isMageBuild(player)))
			{
				topUp(player, potion.itemId(), potion.count());
			}
		}
		
		if (RLAgentEnvironmentConfig.RL_AUTO_ENABLE_SHOTS && (shotCount > 0) && (weapon != null) && (player.getActiveWeaponItem() != player.getFistsWeaponItem()))
		{
			for (int oldShot : new ArrayList<>(player.getAutoSoulShot()))
			{
				if ((oldShot != shotId) && !ALL_BEAST_SHOTS.contains(oldShot))
				{
					player.removeAutoSoulShot(oldShot);
					player.sendPacket(new ExAutoSoulShot(oldShot, 0));
				}
			}
			player.addAutoSoulShot(shotId);
			player.sendPacket(new ExAutoSoulShot(shotId, 1));
			if (summoner)
			{
				for (int beastShot : BEAST_SHOTS)
				{
					player.addAutoSoulShot(beastShot);
					player.sendPacket(new ExAutoSoulShot(beastShot, 1));
				}
			}
			player.rechargeShots(true, true);
		}
	}
	
	private static void topUp(Player player, int itemId, int wanted)
	{
		if ((wanted <= 0) || (ItemData.getInstance().getTemplate(itemId) == null))
		{
			return;
		}
		
		final int missing = wanted - player.getInventory().getInventoryItemCount(itemId, -1);
		if (missing > 0)
		{
			player.addItem(ItemProcessType.REWARD, itemId, missing, null, false);
		}
	}
	
	// --------------------------------------------------------------------------------
	// Farm zones
	// --------------------------------------------------------------------------------
	
	private void teleportToFarmZone(Player player)
	{
		Location destination = RLAgentEnvironmentConfig.RL_FARM_AUTO ? _farmAreas.pickSpot(player) : null;
		if (destination == null)
		{
			final Location anchor = farmZoneAnchor(player);
			if (anchor == null)
			{
				return;
			}
			
			final int spread = RLAgentEnvironmentConfig.RL_FARM_ZONE_SPREAD;
			destination = new Location(anchor.getX() + Rnd.get(-spread, spread), anchor.getY() + Rnd.get(-spread, spread), anchor.getZ());
		}
		teleportTo(player, destination);
	}
	
	private static void teleportTo(Player player, Location destination)
	{
		player.teleToLocation(destination, false);
		
		// A headless client may never send Appearing; finish the teleport for it.
		ThreadPool.schedule(() ->
		{
			if (player.isOnline() && player.isTeleporting())
			{
				player.onTeleported();
			}
		}, TELEPORT_FALLBACK_DELAY);
	}
	
	private static Location farmZoneAnchor(Player player)
	{
		for (FarmZone zone : RLAgentEnvironmentConfig.RL_FARM_ZONES)
		{
			if ((player.getLevel() >= zone.minLevel()) && (player.getLevel() <= zone.maxLevel()))
			{
				if (zone.useStartPoint() || zone.anchors().isEmpty())
				{
					return player.getTemplate().getCreationPoint();
				}
				return zone.anchors().get(Rnd.get(zone.anchors().size()));
			}
		}
		return null;
	}
	
	public static RLAgentManager getInstance()
	{
		return SingletonHolder.INSTANCE;
	}
	
	private static class SingletonHolder
	{
		protected static final RLAgentManager INSTANCE = new RLAgentManager();
	}
}
